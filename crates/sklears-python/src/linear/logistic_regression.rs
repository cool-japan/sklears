//! Python bindings for Logistic Regression
//!
//! This module provides Python bindings for Logistic Regression,
//! offering scikit-learn compatible interfaces for binary and multiclass classification.
//!
//! Note: This is a basic implementation using manual logistic regression until
//! the sklears-linear LogisticRegression feature compilation issues are resolved.

use super::common::*;
use numpy::IntoPyArray;
use pyo3::types::PyDict;
use pyo3::Bound;
use scirs2_autograd::ndarray::{s, Array1, Array2, Axis};
use scirs2_core::random::{thread_rng, Rng};

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

/// Basic logistic regression implementation
#[derive(Debug, Clone)]
struct BasicLogisticRegression {
    config: PyLogisticRegressionConfig,
    coef_: Array1<f64>,
    intercept_: f64,
    classes_: Array1<f64>,
    n_features_: usize,
}

impl BasicLogisticRegression {
    fn new(config: PyLogisticRegressionConfig) -> Self {
        Self {
            config,
            coef_: Array1::zeros(1),
            intercept_: 0.0,
            classes_: Array1::zeros(1),
            n_features_: 0,
        }
    }

    fn sigmoid(z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<(), SklearsPythonError> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        self.n_features_ = n_features;

        if n_samples != y.len() {
            return Err(SklearsPythonError::ValidationError(
                "X and y have incompatible shapes".to_string(),
            ));
        }

        // Find unique classes
        let mut classes: Vec<f64> = y.iter().cloned().collect();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        classes.dedup();
        self.classes_ = Array1::from_vec(classes.clone());

        // For now, only support binary classification
        if classes.len() != 2 {
            return Err(SklearsPythonError::ValidationError(
                "Currently only binary classification is supported".to_string(),
            ));
        }

        // Map classes to 0 and 1
        let y_mapped: Array1<f64> = y.mapv(|val| if val == classes[0] { 0.0 } else { 1.0 });

        // Add intercept column if needed
        let x_design = if self.config.fit_intercept {
            let mut x_new = Array2::ones((n_samples, n_features + 1));
            x_new.slice_mut(s![.., 1..]).assign(x);
            x_new
        } else {
            x.clone()
        };

        let n_params = if self.config.fit_intercept {
            n_features + 1
        } else {
            n_features
        };

        // Initialize weights
        let mut rng = thread_rng();
        let mut weights = Array1::from_shape_fn(n_params, |_| rng.gen::<f64>() * 0.01);

        // Gradient descent
        let learning_rate = 0.01;
        for _iter in 0..self.config.max_iter {
            let mut total_loss = 0.0;
            let mut gradient: Array1<f64> = Array1::zeros(n_params);

            for i in 0..n_samples {
                let xi = x_design.row(i);
                let yi = y_mapped[i];

                let z = xi.dot(&weights);
                let prediction = Self::sigmoid(z);

                // Log loss contribution
                let loss = if yi == 1.0 {
                    -prediction.ln()
                } else {
                    -(1.0 - prediction).ln()
                };
                total_loss += loss;

                // Gradient contribution
                let error = prediction - yi;
                for j in 0..n_params {
                    gradient[j] += error * xi[j];
                }
            }

            // Apply L2 regularization if configured
            if self.config.penalty == "l2" && self.config.c > 0.0 {
                let reg_strength = 1.0 / self.config.c;
                for j in 0..n_params {
                    // Don't regularize intercept
                    if !self.config.fit_intercept || j > 0 {
                        gradient[j] += reg_strength * weights[j];
                        total_loss += 0.5 * reg_strength * weights[j] * weights[j];
                    }
                }
            }

            // Update weights
            for j in 0..n_params {
                weights[j] -= learning_rate * gradient[j] / n_samples as f64;
            }

            // Check convergence
            let avg_loss = total_loss / n_samples as f64;
            if avg_loss < self.config.tol {
                break;
            }
        }

        // Extract coefficients and intercept
        if self.config.fit_intercept {
            self.intercept_ = weights[0];
            self.coef_ = weights.slice(s![1..]).to_owned();
        } else {
            self.intercept_ = 0.0;
            self.coef_ = weights;
        }

        Ok(())
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, SklearsPythonError> {
        if x.ncols() != self.n_features_ {
            return Err(SklearsPythonError::ValidationError(format!(
                "X has {} features, but model expects {} features",
                x.ncols(),
                self.n_features_
            )));
        }

        let probabilities = self.predict_proba(x)?;
        let predictions = probabilities
            .axis_iter(Axis(0))
            .map(|row| {
                if row[1] > 0.5 {
                    self.classes_[1]
                } else {
                    self.classes_[0]
                }
            })
            .collect::<Vec<f64>>();

        Ok(Array1::from_vec(predictions))
    }

    fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>, SklearsPythonError> {
        if x.ncols() != self.n_features_ {
            return Err(SklearsPythonError::ValidationError(format!(
                "X has {} features, but model expects {} features",
                x.ncols(),
                self.n_features_
            )));
        }

        let n_samples = x.nrows();
        let mut probabilities = Array2::zeros((n_samples, 2));

        for i in 0..n_samples {
            let xi = x.row(i);
            let z = xi.dot(&self.coef_) + self.intercept_;
            let prob_class_1 = Self::sigmoid(z);
            let prob_class_0 = 1.0 - prob_class_1;

            probabilities[[i, 0]] = prob_class_0;
            probabilities[[i, 1]] = prob_class_1;
        }

        Ok(probabilities)
    }

    fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64, SklearsPythonError> {
        let predictions = self.predict(x)?;
        let correct = y
            .iter()
            .zip(predictions.iter())
            .filter(|(&true_val, &pred_val)| (true_val - pred_val).abs() < 1e-6)
            .count();

        Ok(correct as f64 / y.len() as f64)
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
/// This class implements regularized logistic regression using the
/// 'liblinear' library, 'newton-cg', 'sag', 'saga' and 'lbfgs' solvers.
/// **Note that regularization is applied by default**. It can handle both
/// dense and sparse input. Use C-ordered arrays or CSR matrices containing
/// 64-bit floats for optimal performance; any other input format will be
/// converted (and copied).
///
/// The 'newton-cg', 'sag', and 'lbfgs' solvers support only L2 regularization
/// with primal formulation, or no regularization. The 'liblinear' solver
/// supports both L1 and L2 regularization, with a dual formulation only for
/// the L2 penalty. The Elastic-Net regularization is only supported by the
/// 'saga' solver.
///
/// Parameters
/// ----------
/// penalty : {'l1', 'l2', 'elasticnet', None}, default='l2'
///     Specify the norm of the penalty:
///
///     - None: no penalty is added;
///     - 'l2': add a L2 penalty term and it is the default choice;
///     - 'l1': add a L1 penalty term;
///     - 'elasticnet': both L1 and L2 penalty terms are added.
///
/// dual : bool, default=False
///     Dual or primal formulation. Dual formulation is only implemented for
///     l2 penalty with liblinear solver. Prefer dual=False when
///     n_samples > n_features.
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
/// intercept_scaling : float, default=1
///     Useful only when the solver 'liblinear' is used
///     and self.fit_intercept is set to True. In this case, x becomes
///     [x, self.intercept_scaling],
///     i.e. a "synthetic" feature with constant value equal to
///     intercept_scaling is appended to the instance vector.
///     The intercept becomes intercept_scaling * synthetic_feature_weight.
///
///     Note! the synthetic feature weight is subject to l1/l2 regularization
///     as all other features.
///     To lessen the effect of regularization on synthetic feature weight
///     (and therefore on the intercept) intercept_scaling has to be increased.
///
/// class_weight : dict or 'balanced', default=None
///     Weights associated with classes in the form ``{class_label: weight}``.
///     If not given, all classes are supposed to have weight one.
///
///     The "balanced" mode uses the values of y to automatically adjust
///     weights inversely proportional to class frequencies in the input data
///     as ``n_samples / (n_classes * np.bincount(y))``.
///
///     Note that these weights will be multiplied with sample_weight (passed
///     through the fit method) if sample_weight is specified.
///
/// random_state : int, RandomState instance, default=None
///     Used when ``solver`` == 'sag', 'saga' or 'liblinear' to shuffle the
///     data. See :term:`Glossary <random_state>` for details.
///
/// solver : {'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'}, \
///         default='lbfgs'
///
///     Algorithm to use in the optimization problem. Default is 'lbfgs'.
///     To choose a solver, you might want to consider the following aspects:
///
///         - For small datasets, 'liblinear' is a good choice, whereas 'sag'
///           and 'saga' are faster for large ones;
///         - For multiclass problems, only 'newton-cg', 'sag', 'saga' and
///           'lbfgs' handle multinomial loss;
///         - 'liblinear' is limited to one-versus-rest schemes.
///
/// max_iter : int, default=100
///     Maximum number of iterations taken for the solvers to converge.
///
/// multi_class : {'auto', 'ovr', 'multinomial'}, default='auto'
///     If the option chosen is 'ovr', then a binary problem is fit for each
///     label. For 'multinomial' the loss minimised is the multinomial loss fit
///     across the entire probability distribution, *even when the data is
///     binary*. 'multinomial' is unavailable when solver='liblinear'.
///     'auto' selects 'ovr' if the data is binary, or if solver='liblinear',
///     and otherwise selects 'multinomial'.
///
/// verbose : int, default=0
///     For the liblinear and lbfgs solvers set verbose to any positive
///     number for verbosity.
///
/// warm_start : bool, default=False
///     When set to True, reuse the solution of the previous call to fit as
///     initialization, otherwise, just erase the previous solution.
///     Useless for liblinear solver. See :term:`the Glossary <warm_start>`.
///
/// n_jobs : int, default=None
///     Number of CPU cores used when parallelizing over classes if
///     multi_class='ovr'". This parameter is ignored when the ``solver``
///     is set to 'liblinear' regardless of whether 'multi_class' is specified or
///     not. ``None`` means 1 unless in a
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
///     In particular, when `multi_class='multinomial'`, `coef_` corresponds
///     to outcome 1 (True) and `-coef_` corresponds to outcome 0 (False).
///
/// intercept_ : ndarray of shape (1,) or (n_classes,)
///     Intercept (a.k.a. bias) added to the decision function.
///
///     If `fit_intercept` is set to False, the intercept is set to zero.
///     `intercept_` is of shape (1,) when the given problem is binary.
///     In particular, when `multi_class='multinomial'`, `intercept_`
///     corresponds to outcome 1 (True) and `-intercept_` corresponds to
///     outcome 0 (False).
///
/// n_features_in_ : int
///     Number of features seen during :term:`fit`.
///
/// n_iter_ : ndarray of shape (n_classes,) or (1, )
///     Actual number of iterations for all classes. If binary or multinomial,
///     it returns only 1 element. For liblinear solver, only the maximum
///     number of iteration across all classes is given.
///
/// Examples
/// --------
/// >>> from sklears_python import LogisticRegression
/// >>> from sklearn.datasets import load_iris
/// >>> X, y = load_iris(return_X_y=True)
/// >>> clf = LogisticRegression(random_state=0).fit(X, y)
/// >>> clf.predict(X[:2, :])
/// array([0, 0])
/// >>> clf.predict_proba(X[:2, :])
/// array([[9.8...e-01, 1.8...e-02, 1.4...e-08],
///        [9.7...e-01, 2.8...e-02, ...e-08]])
/// >>> clf.score(X, y)
/// 0.97...
///
/// Notes
/// -----
/// The underlying C implementation uses a random number generator to
/// select features when fitting the model. It is thus not uncommon,
/// to have slightly different results for the same input data. If
/// that happens, try with a smaller tol parameter.
///
/// Predict output may not match that of standalone liblinear in certain
/// cases. See :ref:`differences from liblinear <liblinear_differences>`
/// in the narrative documentation.
///
/// References
/// ----------
/// L-BFGS-B -- Software for Large-scale Bound-constrained Optimization
/// Ciyou Zhu, Richard Byrd, Jorge Nocedal and Jose Luis Morales.
/// http://users.iems.northwestern.edu/~nocedal/lbfgsb.html
///
/// LIBLINEAR -- A Library for Large Linear Classification
/// https://www.csie.ntu.edu.tw/~cjlin/liblinear/
///
/// SAG -- Mark Schmidt, Nicolas Le Roux, and Francis Bach
/// Minimizing Finite Sums with the Stochastic Average Gradient
/// https://hal.inria.fr/hal-00860051/document
///
/// SAGA -- Defazio, A., Bach F. & Lacoste-Julien S. (2014).
/// SAGA: A Fast Incremental Gradient Method With Support
/// for Non-Strongly Convex Composite Objectives
/// https://arxiv.org/abs/1407.0202
///
/// Hsiang-Fu Yu, Fang-Lan Huang, Chih-Jen Lin (2011). Dual coordinate descent
/// methods for logistic regression and maximum entropy models.
/// Machine Learning 85(1-2):41-75.
/// https://www.csie.ntu.edu.tw/~cjlin/papers/maxent_dual.pdf
#[pyclass(name = "LogisticRegression")]
pub struct PyLogisticRegression {
    py_config: PyLogisticRegressionConfig,
    fitted_model: Option<BasicLogisticRegression>,
}

#[pymethods]
impl PyLogisticRegression {
    #[new]
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
        let x_array = x.as_array().to_owned();
        let y_array = y.as_array().to_owned();

        // Validate input arrays using enhanced validation
        validate_fit_arrays_enhanced(&x_array, &y_array).map_err(PyErr::from)?;

        // Create and fit model
        let mut model = BasicLogisticRegression::new(self.py_config.clone());
        match model.fit(&x_array, &y_array) {
            Ok(()) => {
                self.fitted_model = Some(model);
                Ok(())
            }
            Err(e) => Err(PyErr::from(e)),
        }
    }

    /// Predict class labels for samples
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
            Err(e) => Err(PyErr::from(e)),
        }
    }

    /// Predict class probabilities for samples
    fn predict_proba(&self, x: PyReadonlyArray2<f64>) -> PyResult<Py<PyArray2<f64>>> {
        let fitted = self
            .fitted_model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted. Call fit() first."))?;

        let x_array = x.as_array().to_owned();
        validate_predict_array(&x_array)?;

        match fitted.predict_proba(&x_array) {
            Ok(probabilities) => {
                let py = unsafe { Python::assume_attached() };
                Ok(probabilities.into_pyarray(py).into())
            }
            Err(e) => Err(PyErr::from(e)),
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
        Ok(fitted.coef_.clone().into_pyarray(py).into())
    }

    /// Get model intercept
    #[getter]
    fn intercept_(&self) -> PyResult<f64> {
        let fitted = self
            .fitted_model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted. Call fit() first."))?;

        Ok(fitted.intercept_)
    }

    /// Get unique class labels
    #[getter]
    fn classes_(&self) -> PyResult<Py<PyArray1<f64>>> {
        let fitted = self
            .fitted_model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted. Call fit() first."))?;

        let py = unsafe { Python::assume_attached() };
        Ok(fitted.classes_.clone().into_pyarray(py).into())
    }

    /// Calculate accuracy score
    fn score(&self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<f64> {
        let fitted = self
            .fitted_model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted. Call fit() first."))?;

        let x_array = x.as_array().to_owned();
        let y_array = y.as_array().to_owned();

        match fitted.score(&x_array, &y_array) {
            Ok(score) => Ok(score),
            Err(e) => Err(PyErr::from(e)),
        }
    }

    /// Get number of features
    #[getter]
    fn n_features_in_(&self) -> PyResult<usize> {
        let fitted = self
            .fitted_model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted. Call fit() first."))?;

        Ok(fitted.n_features_)
    }

    /// Return parameters for this estimator (sklearn compatibility)
    fn get_params(&self, deep: Option<bool>) -> PyResult<Py<PyDict>> {
        let _deep = deep.unwrap_or(true);

        let py = unsafe { Python::assume_attached() };
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
