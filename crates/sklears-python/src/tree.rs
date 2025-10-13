//! Python bindings for tree-based algorithms
//!
//! This module provides PyO3-based Python bindings for sklears tree algorithms,
//! including Decision Trees, Random Forest, and Extra Trees.

use crate::utils::{numpy_to_ndarray1, numpy_to_ndarray2};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyList;
use sklears_core::error::{Result as SklearsResult, SklearsError};
use sklears_core::traits::{Fit, Predict};
use sklears_tree::{
    DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreesClassifier, ExtraTreesConfig,
    ExtraTreesRegressor, MaxFeatures, RandomForestClassifier, RandomForestConfig,
    RandomForestRegressor, SamplingStrategy, SplitCriterion,
};

/// Python wrapper for Decision Tree Classifier
#[pyclass(name = "DecisionTreeClassifier")]
pub struct PyDecisionTreeClassifier {
    inner: Option<DecisionTreeClassifier>,
    trained: Option<sklears_tree::TrainedDecisionTreeClassifier>,
}

#[pymethods]
impl PyDecisionTreeClassifier {
    #[new]
    #[pyo3(signature = (
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        class_weight=None,
        ccp_alpha=0.0
    ))]
    fn new(
        criterion: &str,
        splitter: &str,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        min_weight_fraction_leaf: f64,
        max_features: Option<&str>,
        random_state: Option<u64>,
        max_leaf_nodes: Option<usize>,
        min_impurity_decrease: f64,
        class_weight: Option<&str>,
        ccp_alpha: f64,
    ) -> PyResult<Self> {
        let split_criterion = match criterion {
            "gini" => SplitCriterion::Gini,
            "entropy" => SplitCriterion::Entropy,
            "log_loss" => SplitCriterion::LogLoss,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown criterion: {}",
                    criterion
                )))
            }
        };

        let max_features_strategy = match max_features {
            Some("auto") | Some("sqrt") => Some(MaxFeatures::Sqrt),
            Some("log2") => Some(MaxFeatures::Log2),
            None => None,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown max_features: {:?}",
                    max_features
                )))
            }
        };

        // Create the decision tree with configuration
        let mut tree = DecisionTreeClassifier::new();
        tree.criterion = split_criterion;
        tree.max_depth = max_depth;
        tree.min_samples_split = min_samples_split;
        tree.min_samples_leaf = min_samples_leaf;
        tree.min_impurity_decrease = min_impurity_decrease;
        tree.random_state = random_state;

        if let Some(max_features) = max_features_strategy {
            tree.max_features = Some(max_features);
        }

        Ok(Self {
            inner: Some(tree),
            trained: None,
        })
    }

    /// Fit the decision tree classifier
    fn fit(&mut self, x: &PyArray2<f64>, y: &PyArray1<f64>) -> PyResult<()> {
        let x_array = numpy_to_ndarray2(x)?;
        let y_array = numpy_to_ndarray1(y)?;

        // Convert y to integer vector for classification
        let y_int: Vec<usize> = y_array.iter().map(|&val| val as usize).collect();

        let model = self.inner.take().ok_or_else(|| {
            PyRuntimeError::new_err("Model has already been fitted or was not initialized")
        })?;

        match model.fit(&x_array, &y_int) {
            Ok(trained_model) => {
                self.trained = Some(trained_model);
                Ok(())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to fit model: {}",
                e
            ))),
        }
    }

    /// Make predictions using the fitted model
    fn predict<'py>(&self, py: Python<'py>, x: &PyArray2<f64>) -> PyResult<&'py PyArray1<f64>> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before making predictions")
        })?;

        let x_array = numpy_to_ndarray2(x)?;

        match trained_model.predict(&x_array) {
            Ok(predictions) => {
                let predictions_f64: Vec<f64> = predictions.iter().map(|&x| x as f64).collect();
                Ok(PyArray1::from_vec(py, predictions_f64))
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Prediction failed: {}", e))),
        }
    }

    /// Get feature importances
    fn feature_importances_<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray1<f64>> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before accessing feature importances")
        })?;

        match trained_model.feature_importances() {
            Ok(importances) => Ok(importances.into_pyarray(py)),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to compute feature importances: {}",
                e
            ))),
        }
    }

    fn __repr__(&self) -> String {
        if self.trained.is_some() {
            "DecisionTreeClassifier(fitted=True)".to_string()
        } else {
            "DecisionTreeClassifier(fitted=False)".to_string()
        }
    }
}

/// Python wrapper for Decision Tree Regressor
#[pyclass(name = "DecisionTreeRegressor")]
pub struct PyDecisionTreeRegressor {
    inner: Option<DecisionTreeRegressor>,
    trained: Option<sklears_tree::TrainedDecisionTreeRegressor>,
}

#[pymethods]
impl PyDecisionTreeRegressor {
    #[new]
    #[pyo3(signature = (
        criterion="squared_error",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        ccp_alpha=0.0
    ))]
    fn new(
        criterion: &str,
        splitter: &str,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        min_weight_fraction_leaf: f64,
        max_features: Option<&str>,
        random_state: Option<u64>,
        max_leaf_nodes: Option<usize>,
        min_impurity_decrease: f64,
        ccp_alpha: f64,
    ) -> PyResult<Self> {
        let split_criterion = match criterion {
            "squared_error" | "mse" => SplitCriterion::SquaredError,
            "mae" | "absolute_error" => SplitCriterion::MAE,
            "poisson" => SplitCriterion::Poisson,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown criterion: {}",
                    criterion
                )))
            }
        };

        let max_features_strategy = match max_features {
            Some("auto") | Some("sqrt") => Some(MaxFeatures::Sqrt),
            Some("log2") => Some(MaxFeatures::Log2),
            None => None,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown max_features: {:?}",
                    max_features
                )))
            }
        };

        // Create the decision tree with configuration
        let mut tree = DecisionTreeRegressor::new();
        tree.criterion = split_criterion;
        tree.max_depth = max_depth;
        tree.min_samples_split = min_samples_split;
        tree.min_samples_leaf = min_samples_leaf;
        tree.min_impurity_decrease = min_impurity_decrease;
        tree.random_state = random_state;

        if let Some(max_features) = max_features_strategy {
            tree.max_features = Some(max_features);
        }

        Ok(Self {
            inner: Some(tree),
            trained: None,
        })
    }

    /// Fit the decision tree regressor
    fn fit(&mut self, x: &PyArray2<f64>, y: &PyArray1<f64>) -> PyResult<()> {
        let x_array = numpy_to_ndarray2(x)?;
        let y_array = numpy_to_ndarray1(y)?;

        let model = self.inner.take().ok_or_else(|| {
            PyRuntimeError::new_err("Model has already been fitted or was not initialized")
        })?;

        match model.fit(&x_array, &y_array) {
            Ok(trained_model) => {
                self.trained = Some(trained_model);
                Ok(())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to fit model: {}",
                e
            ))),
        }
    }

    /// Make predictions using the fitted model
    fn predict<'py>(&self, py: Python<'py>, x: &PyArray2<f64>) -> PyResult<&'py PyArray1<f64>> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before making predictions")
        })?;

        let x_array = numpy_to_ndarray2(x)?;

        match trained_model.predict(&x_array) {
            Ok(predictions) => Ok(predictions.into_pyarray(py)),
            Err(e) => Err(PyRuntimeError::new_err(format!("Prediction failed: {}", e))),
        }
    }

    /// Get feature importances
    fn feature_importances_<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray1<f64>> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before accessing feature importances")
        })?;

        match trained_model.feature_importances() {
            Ok(importances) => Ok(importances.into_pyarray(py)),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to compute feature importances: {}",
                e
            ))),
        }
    }

    fn __repr__(&self) -> String {
        if self.trained.is_some() {
            "DecisionTreeRegressor(fitted=True)".to_string()
        } else {
            "DecisionTreeRegressor(fitted=False)".to_string()
        }
    }
}

/// Python wrapper for Random Forest Classifier
#[pyclass(name = "RandomForestClassifier")]
pub struct PyRandomForestClassifier {
    inner: Option<RandomForestClassifier>,
    trained: Option<sklears_tree::TrainedRandomForestClassifier>,
}

#[pymethods]
impl PyRandomForestClassifier {
    #[new]
    #[pyo3(signature = (
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=true,
        oob_score=false,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=false,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None
    ))]
    fn new(
        n_estimators: usize,
        criterion: &str,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        min_weight_fraction_leaf: f64,
        max_features: &str,
        max_leaf_nodes: Option<usize>,
        min_impurity_decrease: f64,
        bootstrap: bool,
        oob_score: bool,
        n_jobs: Option<i32>,
        random_state: Option<u64>,
        verbose: i32,
        warm_start: bool,
        class_weight: Option<&str>,
        ccp_alpha: f64,
        max_samples: Option<f64>,
    ) -> PyResult<Self> {
        let split_criterion = match criterion {
            "gini" => SplitCriterion::Gini,
            "entropy" => SplitCriterion::Entropy,
            "log_loss" => SplitCriterion::LogLoss,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown criterion: {}",
                    criterion
                )))
            }
        };

        let max_features_strategy = match max_features {
            "auto" | "sqrt" => MaxFeatures::Sqrt,
            "log2" => MaxFeatures::Log2,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown max_features: {}",
                    max_features
                )))
            }
        };

        let config = RandomForestConfig {
            n_estimators,
            criterion: split_criterion,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features: max_features_strategy,
            bootstrap,
            random_state,
            n_jobs: n_jobs.map(|j| j as usize),
            ..Default::default()
        };

        Ok(Self {
            inner: Some(RandomForestClassifier::new(config)),
            trained: None,
        })
    }

    /// Fit the random forest classifier
    fn fit(&mut self, x: &PyArray2<f64>, y: &PyArray1<f64>) -> PyResult<()> {
        let x_array = numpy_to_ndarray2(x)?;
        let y_array = numpy_to_ndarray1(y)?;

        // Convert y to integer vector for classification
        let y_int: Vec<usize> = y_array.iter().map(|&val| val as usize).collect();

        let model = self.inner.take().ok_or_else(|| {
            PyRuntimeError::new_err("Model has already been fitted or was not initialized")
        })?;

        match model.fit(&x_array, &y_int) {
            Ok(trained_model) => {
                self.trained = Some(trained_model);
                Ok(())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to fit model: {}",
                e
            ))),
        }
    }

    /// Make predictions using the fitted model
    fn predict<'py>(&self, py: Python<'py>, x: &PyArray2<f64>) -> PyResult<&'py PyArray1<f64>> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before making predictions")
        })?;

        let x_array = numpy_to_ndarray2(x)?;

        match trained_model.predict(&x_array) {
            Ok(predictions) => {
                let predictions_f64: Vec<f64> = predictions.iter().map(|&x| x as f64).collect();
                Ok(PyArray1::from_vec(py, predictions_f64))
            }
            Err(e) => Err(PyRuntimeError::new_err(format!("Prediction failed: {}", e))),
        }
    }

    /// Get feature importances
    fn feature_importances_<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray1<f64>> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before accessing feature importances")
        })?;

        match trained_model.feature_importances() {
            Ok(importances) => Ok(importances.into_pyarray(py)),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to compute feature importances: {}",
                e
            ))),
        }
    }

    fn __repr__(&self) -> String {
        if self.trained.is_some() {
            "RandomForestClassifier(fitted=True)".to_string()
        } else {
            "RandomForestClassifier(fitted=False)".to_string()
        }
    }
}

/// Python wrapper for Random Forest Regressor
#[pyclass(name = "RandomForestRegressor")]
pub struct PyRandomForestRegressor {
    inner: Option<RandomForestRegressor>,
    trained: Option<sklears_tree::TrainedRandomForestRegressor>,
}

#[pymethods]
impl PyRandomForestRegressor {
    #[new]
    #[pyo3(signature = (
        n_estimators=100,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=true,
        oob_score=false,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=false,
        ccp_alpha=0.0,
        max_samples=None
    ))]
    fn new(
        n_estimators: usize,
        criterion: &str,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        min_weight_fraction_leaf: f64,
        max_features: f64,
        max_leaf_nodes: Option<usize>,
        min_impurity_decrease: f64,
        bootstrap: bool,
        oob_score: bool,
        n_jobs: Option<i32>,
        random_state: Option<u64>,
        verbose: i32,
        warm_start: bool,
        ccp_alpha: f64,
        max_samples: Option<f64>,
    ) -> PyResult<Self> {
        let split_criterion = match criterion {
            "squared_error" | "mse" => SplitCriterion::SquaredError,
            "mae" | "absolute_error" => SplitCriterion::MAE,
            "poisson" => SplitCriterion::Poisson,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown criterion: {}",
                    criterion
                )))
            }
        };

        let max_features_strategy = if max_features == 1.0 {
            MaxFeatures::All
        } else {
            MaxFeatures::Fraction(max_features)
        };

        let config = RandomForestConfig {
            n_estimators,
            criterion: split_criterion,
            max_depth,
            min_samples_split,
            min_samples_leaf,
            max_features: max_features_strategy,
            bootstrap,
            random_state,
            n_jobs: n_jobs.map(|j| j as usize),
            ..Default::default()
        };

        Ok(Self {
            inner: Some(RandomForestRegressor::new(config)),
            trained: None,
        })
    }

    /// Fit the random forest regressor
    fn fit(&mut self, x: &PyArray2<f64>, y: &PyArray1<f64>) -> PyResult<()> {
        let x_array = numpy_to_ndarray2(x)?;
        let y_array = numpy_to_ndarray1(y)?;

        let model = self.inner.take().ok_or_else(|| {
            PyRuntimeError::new_err("Model has already been fitted or was not initialized")
        })?;

        match model.fit(&x_array, &y_array) {
            Ok(trained_model) => {
                self.trained = Some(trained_model);
                Ok(())
            }
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to fit model: {}",
                e
            ))),
        }
    }

    /// Make predictions using the fitted model
    fn predict<'py>(&self, py: Python<'py>, x: &PyArray2<f64>) -> PyResult<&'py PyArray1<f64>> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before making predictions")
        })?;

        let x_array = numpy_to_ndarray2(x)?;

        match trained_model.predict(&x_array) {
            Ok(predictions) => Ok(predictions.into_pyarray(py)),
            Err(e) => Err(PyRuntimeError::new_err(format!("Prediction failed: {}", e))),
        }
    }

    /// Get feature importances
    fn feature_importances_<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray1<f64>> {
        let trained_model = self.trained.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("Model must be fitted before accessing feature importances")
        })?;

        match trained_model.feature_importances() {
            Ok(importances) => Ok(importances.into_pyarray(py)),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to compute feature importances: {}",
                e
            ))),
        }
    }

    fn __repr__(&self) -> String {
        if self.trained.is_some() {
            "RandomForestRegressor(fitted=True)".to_string()
        } else {
            "RandomForestRegressor(fitted=False)".to_string()
        }
    }
}
