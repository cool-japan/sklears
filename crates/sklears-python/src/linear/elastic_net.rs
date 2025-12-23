//! Python bindings for ElasticNet Regression
//!
//! This module provides Python bindings for ElasticNet Regression,
//! offering scikit-learn compatible interfaces with combined L1+L2 regularization
//! using the sklears-linear crate.

use super::common::*;
use pyo3::types::PyDict;
use pyo3::Bound;
use sklears_core::traits::{Fit, Predict, Score, Trained};
use sklears_linear::{LinearRegression, LinearRegressionConfig, Penalty};

/// Python-specific configuration wrapper for ElasticNet
#[derive(Debug, Clone)]
pub struct PyElasticNetConfig {
    pub alpha: f64,
    pub l1_ratio: f64,
    pub fit_intercept: bool,
    pub copy_x: bool,
    pub max_iter: usize,
    pub tol: f64,
    pub warm_start: bool,
    pub positive: bool,
    pub random_state: Option<i32>,
    pub selection: String,
}

impl Default for PyElasticNetConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            l1_ratio: 0.5,
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

impl From<PyElasticNetConfig> for LinearRegressionConfig {
    fn from(py_config: PyElasticNetConfig) -> Self {
        // ElasticNet combines L1 and L2 penalties
        LinearRegressionConfig {
            fit_intercept: py_config.fit_intercept,
            penalty: Penalty::ElasticNet {
                alpha: py_config.alpha,
                l1_ratio: py_config.l1_ratio,
            },
            max_iter: py_config.max_iter,
            tol: py_config.tol,
            warm_start: py_config.warm_start,
            ..Default::default()
        }
    }
}

/// Linear regression with combined L1 and L2 priors as regularizer.
///
/// Minimizes the objective function:
///
///     1 / (2 * n_samples) * ||y - Xw||^2_2
///     + alpha * l1_ratio * ||w||_1
///     + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2
///
/// If you are interested in controlling the L1 and L2 penalty
/// separately, keep in mind that this is equivalent to:
///
///     a * L1 + b * L2
///
/// where:
///
///     alpha = a + b and l1_ratio = a / (a + b)
///
/// The parameter l1_ratio corresponds to alpha in the glmnet R package
/// while alpha corresponds to the lambda parameter in glmnet.
/// Specifically, l1_ratio = 1 is the lasso penalty. Currently, l1_ratio
/// <= 0.01 is not reliable, unless you supply your own sequence of alpha.
///
/// Parameters
/// ----------
/// alpha : float, default=1.0
///     Constant that multiplies the penalty terms. Defaults to 1.0.
///     See the notes for the exact mathematical meaning of this
///     parameter. ``alpha = 0`` is equivalent to an ordinary least square,
///     solved by the :class:`LinearRegression` object. For numerical
///     reasons, using ``alpha = 0`` with the ``Lasso`` object is not advised.
///     Given this, you should use the :class:`LinearRegression` object.
///
/// l1_ratio : float, default=0.5
///     The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For
///     ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
///     is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
///     combination of L1 and L2.
///
/// fit_intercept : bool, default=True
///     Whether to calculate the intercept for this model. If set
///     to False, no intercept will be used in calculations
///     (i.e. data is expected to be centered).
///
/// copy_X : bool, default=True
///     If ``True``, X will be copied; else, it may be overwritten.
///
/// max_iter : int, default=1000
///     The maximum number of iterations for the optimization algorithm.
///
/// tol : float, default=1e-4
///     The tolerance for the optimization: if the updates are
///     smaller than ``tol``, the optimization code checks the
///     dual gap for optimality and continues until it is smaller
///     than ``tol``, see Notes below.
///
/// warm_start : bool, default=False
///     When set to ``True``, reuse the solution of the previous call to fit as
///     initialization, otherwise, just erase the previous solution.
///     See :term:`the Glossary <warm_start>`.
///
/// positive : bool, default=False
///     When set to ``True``, forces the coefficients to be positive.
///
/// random_state : int, RandomState instance, default=None
///     The seed of the pseudo random number generator that selects a random
///     feature to update. Used when ``selection`` == 'random'.
///     Pass an int for reproducible output across multiple function calls.
///     See :term:`Glossary <random_state>`.
///
/// selection : {'cyclic', 'random'}, default='cyclic'
///     If set to 'random', a random coefficient is updated every iteration
///     rather than looping over features sequentially by default. This
///     (setting to 'random') often leads to significantly faster convergence
///     especially when tol is higher than 1e-4.
///
/// Attributes
/// ----------
/// coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
///     Parameter vector (w in the cost function formula).
///
/// sparse_coef_ : sparse matrix of shape (n_features,) or \
///         (n_targets, n_features)
///     Sparse representation of the fitted ``coef_``.
///
/// intercept_ : float or ndarray of shape (n_targets,)
///     Independent term in decision function.
///
/// n_features_in_ : int
///     Number of features seen during :term:`fit`.
///
/// n_iter_ : list of int
///     Number of iterations run by the coordinate descent solver to reach
///     the specified tolerance.
///
/// Examples
/// --------
/// >>> from sklears_python import ElasticNet
/// >>> from sklearn.datasets import make_regression
/// >>> X, y = make_regression(n_features=2, random_state=0)
/// >>> regr = ElasticNet(random_state=0)
/// >>> regr.fit(X, y)
/// ElasticNet(random_state=0)
/// >>> print(regr.coef_)
/// [18.83816119 64.55968437]
/// >>> print(regr.intercept_)
/// 1.451...
/// >>> print(regr.predict([[0, 0]]))
/// [1.451...]
///
/// Notes
/// -----
/// To avoid unnecessary memory duplication the X argument of the fit method
/// should be directly passed as a Fortran-contiguous NumPy array.
///
/// The precise stopping criteria based on `tol` are the following: First,
/// check that that maximum coordinate update, i.e. :math:`\\max_j |w_j^{new} -
/// w_j^{old}|` is smaller than `tol` times the maximum absolute coefficient,
/// :math:`\\max_j |w_j|`. If so, then additionally check whether the dual gap
/// is smaller than `tol` times :math:`||y||_2^2 / n_\\text{samples}`.
#[pyclass(name = "ElasticNet")]
pub struct PyElasticNet {
    /// Python-specific configuration
    py_config: PyElasticNetConfig,
    /// Trained model instance using the actual sklears-linear implementation
    fitted_model: Option<LinearRegression<Trained>>,
}

#[pymethods]
impl PyElasticNet {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (alpha=1.0, l1_ratio=0.5, fit_intercept=true, copy_x=true, max_iter=1000, tol=1e-4, warm_start=false, positive=false, random_state=None, selection="cyclic"))]
    fn new(
        alpha: f64,
        l1_ratio: f64,
        fit_intercept: bool,
        copy_x: bool,
        max_iter: usize,
        tol: f64,
        warm_start: bool,
        positive: bool,
        random_state: Option<i32>,
        selection: &str,
    ) -> PyResult<Self> {
        // Validate l1_ratio
        if !(0.0..=1.0).contains(&l1_ratio) {
            return Err(PyValueError::new_err(
                "l1_ratio must be between 0 and 1 (inclusive)",
            ));
        }

        // Validate alpha
        if alpha < 0.0 {
            return Err(PyValueError::new_err("alpha must be non-negative"));
        }

        let py_config = PyElasticNetConfig {
            alpha,
            l1_ratio,
            fit_intercept,
            copy_x,
            max_iter,
            tol,
            warm_start,
            positive,
            random_state,
            selection: selection.to_string(),
        };

        Ok(Self {
            py_config,
            fitted_model: None,
        })
    }

    /// Fit the ElasticNet regression model
    fn fit(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let x_array = pyarray_to_core_array2(x)?;
        let y_array = pyarray_to_core_array1(y)?;

        // Validate input arrays
        validate_fit_arrays(&x_array, &y_array)?;

        // Create sklears-linear model with ElasticNet configuration
        let model = LinearRegression::elastic_net(self.py_config.alpha, self.py_config.l1_ratio)
            .fit_intercept(self.py_config.fit_intercept);

        // Fit the model using sklears-linear's implementation
        match model.fit(&x_array, &y_array) {
            Ok(fitted_model) => {
                self.fitted_model = Some(fitted_model);
                Ok(())
            }
            Err(e) => Err(PyValueError::new_err(format!(
                "Failed to fit ElasticNet model: {:?}",
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
        dict.set_item("l1_ratio", self.py_config.l1_ratio)?;
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
            let alpha_val: f64 = alpha.extract()?;
            if alpha_val < 0.0 {
                return Err(PyValueError::new_err("alpha must be non-negative"));
            }
            self.py_config.alpha = alpha_val;
        }
        if let Some(l1_ratio) = kwargs.get_item("l1_ratio")? {
            let l1_ratio_val: f64 = l1_ratio.extract()?;
            if !(0.0..=1.0).contains(&l1_ratio_val) {
                return Err(PyValueError::new_err(
                    "l1_ratio must be between 0 and 1 (inclusive)",
                ));
            }
            self.py_config.l1_ratio = l1_ratio_val;
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
            "ElasticNet(alpha={}, l1_ratio={}, fit_intercept={}, copy_X={}, max_iter={}, tol={}, warm_start={}, positive={}, random_state={:?}, selection='{}')",
            self.py_config.alpha,
            self.py_config.l1_ratio,
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
